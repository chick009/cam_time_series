import React from 'react'
import { useNavigate } from 'react-router-dom';
const TableList = (props) => {
    const navigate = useNavigate();

    const UpdateHandler = (e) => {
        e.preventDefault()
        navigate('/update')
    }
    
    const CancelHandler = (e) => {
        e.preventDefault()
        
    }
    return(
        <tr>
            <td>{props.title}</td>
            <td>{props.date}</td>
            <td>{props.description}</td>
            <td>{props.presenter}</td>
            <td>{props.price}</td>
            <td>
                <button
                    id={props.key}
                    onClick={UpdateHandler}>Update</button>
            </td>
            <td><button
                    id={props.key}
                    onClick={CancelHandler}>Cancel</button></td>
        </tr>
    )
}

export default TableList;