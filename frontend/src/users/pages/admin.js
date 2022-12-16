import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import TableList from '../components/tablelist'
import Table from 'react-bootstrap/Table';
import { getActiveElement } from '@testing-library/user-event/dist/utils';

const Admin = () => {
    const [tables, setTables] = useState([])
    const navigate = useNavigate()

    const AddEventHandler = (e) => {
        e.preventDefault()
        navigate('/create')
    }

    const cancelHandler = (e) => {

    }
    
    useEffect(() => {
        const fetchData = async () => {
            const url = "http://localhost:5000/api/admin/events"
            const response = await fetch(url)
            const data = await response.json()
            setTables(data.events.slice(0, 10))
        }
        fetchData()
    }, [])
    
    return(
    <>  
        <div>
            To create data, please click <button onClick={AddEventHandler}> Create Data </button>
        </div>

        <hr/>

        <Table striped center bordered hover responsive size="sm">
            <thead centre>New Title</thead>
            {tables.map((obj, idx) => (                
                <TableList key={idx} title={obj.title} date={obj.date} description={obj.description} presenter={obj.presenter} price={obj.price}/>
            ))}
        </Table>
        
        
    </>)
}

export default Admin